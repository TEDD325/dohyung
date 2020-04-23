/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @microservice:  export-client
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/
package org.edgexfoundry.dao.integration.mongo;

import static org.edgexfoundry.test.data.RegistrationData.TEST_NAME;
import static org.edgexfoundry.test.data.RegistrationData.checkTestData;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.ExportRegistrationRepository;
import org.edgexfoundry.domain.export.ExportRegistration;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.RegistrationData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({ RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class })
public class ExportRegistrationRepositoryTest {

	@Autowired
	private ExportRegistrationRepository repos;

	private String id;

	// setup tests add
	@Before
	public void setup() {
		ExportRegistration registration = RegistrationData.newTestInstance();
		repos.save(registration);
		id = registration.getId();
	}

	// setup tests delete all
	@After
	public void cleanup() {
		repos.deleteAll();
	}

	@Test
	public void testFindAll() {
		List<ExportRegistration> regs = repos.findAll();
		assertEquals("Size of registrations is not one", 1, regs.size());
		checkTestData(regs.get(0), id);
	}

	@Test
	public void testFindById() {
		checkTestData(repos.findOne(id), id);
	}

	@Test
	public void testFindByEnabled() {
		List<ExportRegistration> regs = repos.findByEnable(true);
		assertEquals("Size of enabled registrations is not one", 1, regs.size());
		checkTestData(regs.get(0), id);
		List<ExportRegistration> regs2 = repos.findByEnable(false);
		assertTrue("Size of disabled registrations is not empty", regs2.isEmpty());
	}

	@Test
	public void testFindByName() {
		checkTestData(repos.findByName(TEST_NAME), id);
	}

	@Test
	public void testDelete() {
		repos.delete(repos.findOne(id));
		assertTrue("Delete did not remove registrations", repos.findAll().isEmpty());
	}

	@Test
	public void testDeleteById() {
		repos.delete(id);
		assertTrue("Delete by id did not remove registrations", repos.findAll().isEmpty());
	}

	@Test
	public void testUpdate() {
		ExportRegistration reg = repos.findOne(id);
		reg.setEnable(false);
		repos.save(reg);
		ExportRegistration reg2 = repos.findOne(id);
		assertFalse("Update of registration not working", reg2.isEnable());
	}

}
